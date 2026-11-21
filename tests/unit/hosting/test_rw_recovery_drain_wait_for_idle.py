# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-E NT-RW-REC-03 (EX-D7-01) and NT-RW-REC-04 (EX-D4-01): a product drain controller.

REC-03: ``BackgroundActiveWorkController`` over a real ``WorkerRuntime`` implements the hosting
``HostedApplicationActiveWorkController`` port: it counts the in-flight request, waits for it, and a
request after ``stop_intake`` is refused before any state write (so a redelivery after restart runs it
exactly once).

REC-04: the real hosted engine built by ``run_hosted_application`` receives the controller through the
runner seam and its shutdown executor waits for idle before the runtime-stop phase.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.events import TaskEvent
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.hosting import (
    HostedApplicationProfile,
    InstancePolicy,
    RestartPolicy,
    ShutdownPolicy,
    run_hosted_application,
)
from intergrax.hosting.contracts.context import (
    HostedApplicationClock,
    HostedApplicationContext,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.policies import InstanceExclusivityMode
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.runner import _create_instance_guard, _RunnerFactories
from intergrax.hosting.shutdown import (
    HostedApplicationActiveWorkController,
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
    SystemMonotonicClock,
)
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskResult, TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from tests.unit.hosting.engine._fakes import NoopLogger, RecordingPublisher
from tests.unit.runtime.background_execution.reentry_admission_doubles import (
    make_kv_admission_dependencies,
)

pytestmark = [pytest.mark.unit]

_TENANT = "t-rw-rec-drain"
_TASK_NAME = "rw.rec.drain.v1"
_WAIT = 10.0


class _Output(BaseModel):
    done: bool = True


class _RecordingKV(DistributedKVStore):
    """In-memory KV that counts writes, so a refused request can be shown to write nothing."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self.writes = 0

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        del ttl_seconds
        self.writes += 1
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self.writes += 1
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


class _Handler:
    """Records the transport run id of every call; the calls can be held open."""

    def __init__(self, *, hold: threading.Event | None = None) -> None:
        self.hold = hold
        self.entered = threading.Event()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None = None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[_Output]:
        del tenant_id, run_id, idempotency_key, execution_identity
        marker = str(json.loads(payload.decode("utf-8"))["marker"])  # the transport run id of the request
        with self._lock:
            self.calls.append(marker)
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=_WAIT), "handler was never released"
        return ToolExecutionResult.ok(_Output())


class _Events:
    def __init__(self) -> None:
        self.events: list[TaskEvent] = []

    def emit(self, event: TaskEvent) -> None:
        self.events.append(event)


def _runtime(
    kv: _RecordingKV,
    handler: _Handler,
    events: _Events | None = None,
    *,
    idempotency_store: IdempotencyStore | None = None,
) -> WorkerRuntime:
    registry = TaskRegistry()
    registry.register(TaskDefinition(task_name=_TASK_NAME, payload_schema=dict, handler=handler))
    execution_registry = TaskExecutionRegistry()
    registry.bind_execution_registry(execution_registry)
    admission = make_kv_admission_dependencies(kv)
    return WorkerRuntime(
        registry=registry,
        state_store=TaskStateStore(kv_store=kv, provider="kv"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="kv",
        idempotency_store=idempotency_store,
        event_emitter=events,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        identity_persistence=admission.identity_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
    )


def _request(run_id: str) -> TaskRequest:
    payload = json.dumps({"marker": run_id}).encode("utf-8")
    return TaskRequest(
        tenant_id=_TENANT,
        run_id=run_id,
        task_name=_TASK_NAME,
        payload=payload,
        idempotency_key=f"key-{run_id}",
    )


def test_rw_rec_03_product_controller_drains_and_refuses_new_work() -> None:
    from intergrax.background_tasks.active_work import BackgroundActiveWorkController
    from intergrax.background_tasks.worker_runtime import WorkerIntakeClosedError

    kv = _RecordingKV()
    hold = threading.Event()
    handler = _Handler(hold=hold)
    events = _Events()
    ledger = InMemoryIdempotencyStore()
    runtime = _runtime(kv, handler, events, idempotency_store=ledger)
    controller = BackgroundActiveWorkController([runtime], cancel_grace_seconds=0.05)
    assert isinstance(controller, HostedApplicationActiveWorkController)
    assert controller.active_work_count() == 0

    results: list[TaskResult] = []
    worker = threading.Thread(
        target=lambda: results.append(runtime.process_request(_request("run-1"), task_id="run-1"))
    )
    worker.start()
    try:
        assert handler.entered.wait(timeout=_WAIT)
        assert controller.active_work_count() == 1

        async def _drain() -> None:
            await controller.stop_intake()
            waiter = asyncio.ensure_future(controller.wait_for_idle())
            await asyncio.sleep(0.2)
            assert not waiter.done(), "wait_for_idle returned while a request was still running"

            writes_before, events_before = kv.writes, len(events.events)
            with pytest.raises(WorkerIntakeClosedError):
                runtime.process_request(_request("run-late"), task_id="run-late")
            assert (kv.writes, len(events.events)) == (writes_before, events_before), "a refused request wrote state"
            assert controller.active_work_count() == 1

            hold.set()
            await asyncio.wait_for(waiter, timeout=_WAIT)

        asyncio.run(_drain())
    finally:
        hold.set()
        worker.join(timeout=_WAIT)

    assert [r.status for r in results] == [TaskStatus.SUCCEEDED]
    assert controller.active_work_count() == 0
    assert handler.calls == ["run-1"]

    # Restart over the same KV and idempotency ledger: the drained request is not re-run, the refused
    # one (no side effect at the gate) runs exactly once.
    restarted_handler = _Handler()
    restarted = _runtime(kv, restarted_handler, idempotency_store=ledger)
    assert restarted.process_request(_request("run-1"), task_id="run-1").status is TaskStatus.SUCCEEDED
    assert restarted.process_request(_request("run-late"), task_id="run-late").status is TaskStatus.SUCCEEDED
    assert restarted_handler.calls == ["run-late"]
    assert handler.calls + restarted_handler.calls == ["run-1", "run-late"]


class _SystemClock(HostedApplicationClock):
    def now(self) -> datetime:
        return datetime.now(UTC)


class _QuietSignals:
    def install(self) -> None:
        return None

    def restore(self) -> None:
        return None


class _DrainProbeRuntime:
    """Application runtime that starts one held worker request and then asks the host to stop."""

    def __init__(self, worker: WorkerRuntime, handler: _Handler, journal: list[str]) -> None:
        self._worker = worker
        self._handler = handler
        self.journal = journal
        self.results: list[TaskResult] = []
        self.thread: threading.Thread | None = None

    async def start(self, context: HostedApplicationContext) -> None:
        def _work() -> None:
            self.results.append(self._worker.process_request(_request("run-host"), task_id="run-host"))

        self.thread = threading.Thread(target=_work, daemon=True)
        self.thread.start()
        assert await asyncio.to_thread(self._handler.entered.wait, _WAIT)
        context.shutdown.request_shutdown("rw_rec_04_stop")

    async def stop(self, context: HostedApplicationContext) -> None:
        del context
        self.journal.append("runtime_stop")

    async def ready(self, context: HostedApplicationContext) -> bool:
        del context
        return True


def test_rw_rec_04_engine_waits_for_idle_through_runner_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.background_tasks.active_work import BackgroundActiveWorkController

    journal: list[str] = []
    wait_started = threading.Event()

    class _JournalingController(BackgroundActiveWorkController):
        async def wait_for_idle(self) -> None:
            journal.append("wait_for_idle:enter")
            wait_started.set()
            await super().wait_for_idle()
            journal.append("wait_for_idle:exit")

    kv = _RecordingKV()
    hold = threading.Event()
    handler = _Handler(hold=hold)
    worker = _runtime(kv, handler)
    controller = _JournalingController([worker], cancel_grace_seconds=0.05)
    runtime = _DrainProbeRuntime(worker, handler, journal)

    def _release_after_drain_began() -> None:
        if wait_started.wait(timeout=_WAIT):
            time.sleep(0.2)
            journal.append("released")
        hold.set()

    releaser = threading.Thread(target=_release_after_drain_began, daemon=True)
    releaser.start()

    def _paths(definition: HostedApplicationDefinition) -> HostedApplicationPaths:
        return HostedApplicationPaths(
            data_home=(tmp_path / "data" / definition.application_id).resolve(),
            run_directory=(tmp_path / "run").resolve(),
        )

    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _RunnerFactories(
            create_paths=_paths,
            create_clock=_SystemClock,
            create_monotonic_clock=SystemMonotonicClock,
            create_logger=lambda _application_id: NoopLogger(),
            create_event_publisher=RecordingPublisher,
            create_process_identity=lambda clock: HostedApplicationProcessIdentity(
                process_id=os.getpid(),
                started_at=clock.now(),
            ),
            create_instance_guard=_create_instance_guard,
            create_signal_adapter=lambda _control: _QuietSignals(),
        ),
    )
    profile = HostedApplicationProfile(
        application_id="rw_rec_drain_app",
        application_factory=lambda: runtime,
        application_factory_id="tests.unit.hosting.test_rw_recovery_drain_wait_for_idle.runtime",
        restart=RestartPolicy.never(),
        instance=InstancePolicy(exclusivity_mode=InstanceExclusivityMode.MULTI_INSTANCE),
        shutdown=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=_WAIT,
            cancel_timeout_seconds=1.0,
            flush_timeout_seconds=1.0,
        ),
    )

    try:
        result = run_hosted_application(profile, active_work_controller_factory=lambda: controller)
    finally:
        hold.set()
        releaser.join(timeout=_WAIT)
        if runtime.thread is not None:
            runtime.thread.join(timeout=_WAIT)

    assert journal == ["wait_for_idle:enter", "released", "wait_for_idle:exit", "runtime_stop"]
    assert [r.status for r in runtime.results] == [TaskStatus.SUCCEEDED]
    assert handler.calls == ["run-host"]
    terminal = result.attempts[-1].terminal_result
    assert terminal is not None
    execution = terminal.diagnostics.shutdown_execution
    assert execution is not None
    assert (execution.active_work_before, execution.active_work_after) == (1, 0)
    assert execution.timed_out is False and execution.forced is False
    order = [record.phase for record in execution.phase_records]
    phases = {record.phase: record for record in execution.phase_records}
    assert phases[HostedApplicationShutdownPhase.STOP_INTAKE].outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED
    assert phases[HostedApplicationShutdownPhase.DRAIN].outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED
    assert HostedApplicationShutdownPhase.CANCEL not in phases
    assert order.index(HostedApplicationShutdownPhase.DRAIN) < order.index(HostedApplicationShutdownPhase.RUNTIME_STOP)
