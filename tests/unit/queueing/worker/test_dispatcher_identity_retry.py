# © Artur Czarnecki. All rights reserved.

"""Celery dispatcher retry identity semantics (BG-EXEC-2)."""

from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import pytest
from celery import Celery
from pydantic import BaseModel

from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    bootstrap_background_execution,
)
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.tools.execution_models import ToolExecutionResult
from tests.unit.queueing.worker.dispatcher_test_kv import DispatcherTestKVStore

pytestmark = pytest.mark.unit


class _Output(BaseModel):
    value: str = "ok"


def test_celery_path_uses_stable_transport_task_id_for_identity_resolution() -> None:
    kv = DispatcherTestKVStore()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="celery",
        transport_task_id="celery-task-abc",
    )

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


def test_celery_dispatcher_bootstrap_receives_celery_transport_ref() -> None:
    app = Celery("test")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True

    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str],
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[_Output]:
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult.ok(_Output())

    registry.register("demo.task.v1", handler)
    register_dispatcher_task(
        app=app,
        registry=registry,
        kv_store=DispatcherTestKVStore(),
    )

    captured: list[BackgroundTransportExecutionRef] = []

    def _capture_bootstrap(**kwargs):
        captured.append(kwargs["transport_ref"])
        return bootstrap_background_execution(**kwargs)

    with patch(
        "intergrax.queueing.worker.dispatcher.bootstrap_background_execution",
        side_effect=_capture_bootstrap,
    ):
        app.tasks["intergrax.execute"].apply(
            kwargs={
                "logical_task_name": "demo.task.v1",
                "tenant_id": "tenant-a",
                "run_id": "queue-correlation",
                "payload": b"{}",
                "idempotency_key": None,
            }
        )

    assert len(captured) == 1
    assert captured[0].tenant_id == "tenant-a"
    assert captured[0].provider == "celery"
    assert captured[0].transport_task_id
