# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
import uuid
import asyncio
from typing import Tuple

import pytest

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.adapters.threaded_adapter import ThreadedExecutionAdapter
from intergrax.fastapi_core.execution.policies import ExecutionPolicy
from intergrax.fastapi_core.execution.worker_contract import CancellableExecutionWorker, ExecutionWorker
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore


class NoOpCancellableWorker(CancellableExecutionWorker):

    def execute(self, request):
        return

    def cancel(self, run_id: str) -> None:
        pass


def build_request(run_id: str) -> ExecutionRequest:
    """
    Builds minimal valid execution request matching runtime contract.
    """
    return ExecutionRequest(
        run_id=run_id,
        tenant_id="test-tenant",
        user_id="test-user",
        input_payload={},
        metadata={},
    )

def create_pending_run(store: InMemoryRunStore, run_id: str) -> None:
    """
    Creates run in PENDING state so adapter can transition it to RUNNING.
    """
    run = store.create()
    store._runs[run_id] = run  # inject correct ID
    store.update_status(run_id, RunStatus.PENDING)


# ------------------------------------------------------------------
# Runtime graph builder
# ------------------------------------------------------------------


@pytest.fixture
def runtime() -> Tuple[ThreadedExecutionAdapter, DefaultRunService, InMemoryRunStore]:
    """
    Builds real runtime dependency graph:

    Store ↔ RunService ↔ ExecutionAdapter
    """
    store = InMemoryRunStore()

    worker = NoOpCancellableWorker()

    adapter = ThreadedExecutionAdapter(
        worker=worker,
        run_service=None,  # injected later
        policy=ExecutionPolicy(max_retries=2, timeout_seconds=10),
    )

    run_service = DefaultRunService(
        store=store,
        execution_adapter=adapter,
    )

    # Close circular dependency (runtime wiring)
    adapter._run_service = run_service

    return adapter, run_service, store


# ------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------


class FlakyWorker(ExecutionWorker):
    def __init__(self) -> None:
        self.calls: int = 0

    def execute(self, request: ExecutionRequest) -> dict[str, bool]:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("transient failure")
        return {"ok": True}


class PermanentFailWorker(ExecutionWorker):
    def execute(self, request: ExecutionRequest) -> None:
        raise ValueError("permanent error")


class AlwaysTransientWorker(ExecutionWorker):
    def __init__(self) -> None:
        self.calls: int = 0

    def execute(self, request: ExecutionRequest) -> None:
        self.calls += 1
        raise TimeoutError("retryable")


class SlowWorker(ExecutionWorker):
    def execute(self, request: ExecutionRequest) -> dict[str, bool]:
        time.sleep(5)
        return {"ok": True}


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_transient_then_success(runtime) -> None:
    adapter, run_service, store = runtime

    worker = FlakyWorker()
    adapter._worker = worker

    request = build_request(str(uuid.uuid4()))

    create_pending_run(store, request.run_id)
    await adapter.start_execution(request)
    await asyncio.sleep(1)

    run: RunResponse = store.get(request.run_id)

    assert run.status == RunStatus.COMPLETED
    assert worker.calls == 2

@pytest.mark.asyncio
async def test_permanent_no_retry(runtime) -> None:
    adapter, run_service, store = runtime

    worker = PermanentFailWorker()
    adapter._worker = worker

    request = build_request(str(uuid.uuid4()))

    create_pending_run(store, request.run_id)
    await adapter.start_execution(request)
    await asyncio.sleep(0.5)

    run: RunResponse = store.get(request.run_id)

    assert run.status == RunStatus.FAILED

@pytest.mark.asyncio
async def test_retry_budget_exhausted(runtime) -> None:
    adapter, run_service, store = runtime

    worker = AlwaysTransientWorker()
    adapter._worker = worker

    request = build_request(str(uuid.uuid4()))

    create_pending_run(store, request.run_id)
    await adapter.start_execution(request)
    await asyncio.sleep(2)

    run: RunResponse = store.get(request.run_id)

    assert run.status == RunStatus.FAILED
    assert worker.calls == 3  # 1 + 2 retries

@pytest.mark.asyncio
async def test_timeout_during_retry(runtime) -> None:
    adapter, run_service, store = runtime

    worker = SlowWorker()
    adapter._worker = worker

    adapter._policy = ExecutionPolicy(
        max_retries=2,
        timeout_seconds=1,
    )

    request = build_request(str(uuid.uuid4()))

    create_pending_run(store, request.run_id)
    await adapter.start_execution(request)
    await asyncio.sleep(2)

    run: RunResponse = store.get(request.run_id)

    assert run.status == RunStatus.FAILED
