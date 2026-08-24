# © Artur Czarnecki. All rights reserved.

"""BG-EXEC-1 — central background execution bootstrap tests."""

from __future__ import annotations

import json
import base64
from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    BackgroundExecutionTenantMismatchError,
    bootstrap_background_execution,
)
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = pytest.mark.unit


class _Output(BaseModel):
    value: str = "ok"


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


class _TestWorker(BrokerWorkerBase):
    def start(self) -> None:
        raise NotImplementedError


def _build_message(*, task_id: str = "queue-handle-1") -> bytes:
    encoded_payload = base64.b64encode(b"input").decode("ascii")
    message = {
        "task_id": task_id,
        "tenant_id": "tenant-a",
        "run_id": "queue-correlation-run",
        "task_name": "demo.task.v1",
        "payload": encoded_payload,
        "idempotency_key": None,
    }
    return json.dumps(message).encode("utf-8")


def test_bootstrap_mints_canonical_identity() -> None:
    identity = bootstrap_background_execution(transport_tenant_id="tenant-a")
    assert identity.tenant_id == "tenant-a"
    assert str(identity.task_id).startswith("task_")
    assert str(identity.run_id).startswith("run_")
    assert str(identity.attempt_id).startswith("attempt_")


def test_bootstrap_rejects_tenant_mismatch() -> None:
    with pytest.raises(BackgroundExecutionTenantMismatchError):
        bootstrap_background_execution(
            transport_tenant_id="tenant-a",
            task_tenant_id="tenant-b",
        )


def test_bootstrap_accepts_canonical_upstream_task_id() -> None:
    task_id = mint_task_id()
    identity = bootstrap_background_execution(
        transport_tenant_id="tenant-a",
        canonical_task_id=task_id,
    )
    assert identity.task_id == task_id


def test_broker_worker_path_uses_central_bootstrap() -> None:
    kv = _KV()
    registry = TaskExecutionRegistry()
    captured: list[BackgroundExecutionIdentity] = []

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity: BackgroundExecutionIdentity | None = None,
    ) -> ToolExecutionResult[_Output]:
        assert execution_identity is not None
        captured.append(execution_identity)
        assert tenant_id == execution_identity.tenant_id
        assert run_id == str(execution_identity.run_id)
        return ToolExecutionResult.ok(_Output())

    registry.register("demo.task.v1", handler)
    worker = _TestWorker(registry=registry, kv_store=kv, idempotency_store=None)
    fixed = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )

    with patch(
        "intergrax.queueing.providers.broker_worker_base.bootstrap_background_execution",
        return_value=fixed,
    ) as bootstrap_mock:
        worker.process_message(raw_payload=_build_message())

    bootstrap_mock.assert_called_once_with(transport_tenant_id="tenant-a")
    assert captured == [fixed]


def test_worker_runtime_path_uses_central_bootstrap() -> None:
    kv = _KV()
    task_registry = TaskRegistry()
    execution_registry = TaskExecutionRegistry()
    captured: list[BackgroundExecutionIdentity] = []

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None = None,
        execution_identity: BackgroundExecutionIdentity | None = None,
    ):
        assert execution_identity is not None
        captured.append(execution_identity)
        return ToolExecutionResult.ok(_Output())

    task_registry.register(
        TaskDefinition(
            task_name="demo.task.v1",
            payload_schema=dict,
            handler=handler,
        )
    )
    task_registry.bind_execution_registry(execution_registry)
    runtime = WorkerRuntime(
        registry=task_registry,
        state_store=TaskStateStore(kv_store=kv, provider="kafka"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="kafka",
    )
    fixed = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    request = TaskRequest(
        tenant_id="tenant-a",
        run_id="queue-correlation-run",
        task_name="demo.task.v1",
        payload=b"{}",
    )

    with patch(
        "intergrax.background_tasks.worker_runtime.bootstrap_background_execution",
        return_value=fixed,
    ) as bootstrap_mock:
        runtime.process_request(request, task_id="queue-handle-1")

    bootstrap_mock.assert_called_once_with(transport_tenant_id="tenant-a")
    assert captured == [fixed]


def test_identity_propagated_without_re_minting_at_handler() -> None:
    registry = TaskExecutionRegistry()
    seen: dict[str, object] = {}

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity: BackgroundExecutionIdentity | None = None,
    ) -> ToolExecutionResult[_Output]:
        seen["tenant_id"] = tenant_id
        seen["run_id"] = run_id
        seen["execution_identity"] = execution_identity
        return ToolExecutionResult.ok(_Output())

    registry.register("demo.task.v1", handler)
    worker = _TestWorker(registry=registry, kv_store=_KV(), idempotency_store=None)
    fixed = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=TaskId("task_" + "a" * 32),
        run_id=RunId("run_" + "b" * 32),
        attempt_id=AttemptId("attempt_" + "c" * 32),
    )

    with patch(
        "intergrax.queueing.providers.broker_worker_base.bootstrap_background_execution",
        return_value=fixed,
    ):
        worker.process_message(raw_payload=_build_message())

    assert seen["tenant_id"] == fixed.tenant_id
    assert seen["run_id"] == str(fixed.run_id)
    assert seen["execution_identity"] == fixed


def test_nexus_worker_preserves_bootstrap_identity_end_to_end() -> None:
    from echo.echo_agent import EchoAgent
    from intergrax.fastapi_core.execution.models import ExecutionRequest
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
    from intergrax.runtime.task.task import Task, TaskContext
    from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
    from intergrax.runtime.task.worker_payload import encode_execution_request

    agent_registry = AgentRegistry()
    agent_registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(agent_registry)
    fixed = BackgroundExecutionIdentity(
        tenant_id="tenant-a",
        task_id=TaskId("task_" + "d" * 32),
        run_id=RunId("run_" + "e" * 32),
        attempt_id=AttemptId("attempt_" + "f" * 32),
    )
    task = Task(
        tenant_id="tenant-a",
        user_id="user-1",
        message="bootstrap identity",
        context=TaskContext(capability="echo.basic"),
    )
    request = ExecutionRequest(
        run_id="queue-correlation-run",
        tenant_id="tenant-a",
        user_id="user-1",
        input_payload=task_to_execution_payload(task),
    )
    payload = encode_execution_request(request)

    result_payload = runtime.execute_payload(
        payload,
        tenant_id="tenant-a",
        run_id="queue-correlation-run",
        execution_identity=fixed,
    )

    assert result_payload["task_id"] == str(fixed.task_id)
    assert result_payload["run_id"] == str(fixed.run_id)
    active = peek_active_execution_identity()
    assert active is None
