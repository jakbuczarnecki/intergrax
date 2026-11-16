# © Artur Czarnecki. All rights reserved.

"""XD-05 (Host <-> Background): shutdown drains in-flight work; leases are released or fenced.

PDC-D4-R005 and PDC-D7-R014. Integrated version of the lane stand-in in
``d4/test_xd05_drain_background_work.py``: the in-flight work here is real
``WorkerRuntime.process_request`` calls (durable SQLite KV plus the SQLite idempotency ledger from
the canonical SQLite persistence root) and a real ``LongRunningScheduler`` claim, and the real
``HostedApplicationEngine`` shutdown executor drains them.

Only the ``HostedApplicationActiveWorkController`` adapter below is test-side: the product ships the
port but no implementation over ``WorkerRuntime`` (reported as a finding). The adapter holds no
authority; it only counts and awaits the in-flight calls.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.contracts.idempotency_store import ClaimOutcome
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.hosting import (
    HostedApplicationEventType,
    HostedApplicationExitKind,
    HostedApplicationLifecycleState,
    ShutdownPolicy,
)
from intergrax.hosting.shutdown import (
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
)
from intergrax.hosting.supervisor.classification import HostedApplicationExitClassifier
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    validate_background_execution_admission_durability,
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.persistence.sqlite_composition import create_sqlite_runtime_persistence
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from tests.qualification.pdc_x.d4._support import (
    APP_ID,
    JournalRuntime,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)
from tests.qualification.pdc_x.d7._support import ok_task_result, paused_checkpoint

pytestmark = [pytest.mark.integration]

TENANT = "tenant-xd05"
TASK_NAME = "pdc.xd05.background_work"
T0 = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)
_WAIT = 20.0


class _Output(BaseModel):
    done: bool = True


class _Counter:
    """Counts handler invocations per idempotency key and can hold the handler open."""

    def __init__(self, hold: threading.Event | None = None) -> None:
        self.hold = hold
        self.entered = threading.Event()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def handler(self, *, tenant_id, run_id, payload, idempotency_key, execution_identity):  # type: ignore[no-untyped-def]
        with self._lock:
            self.calls.append(str(idempotency_key))
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=_WAIT), "handler was never released"
        return ToolExecutionResult.ok(_Output())


class _WorkerActiveWork:
    """Hosting active-work port over real ``WorkerRuntime.process_request`` calls."""

    def __init__(self, runtime: WorkerRuntime) -> None:
        self._runtime = runtime
        self._tasks: set[asyncio.Task[object]] = set()
        self._intake_open = True
        self.cancelled = 0

    def submit(self, request: TaskRequest, task_id: str) -> asyncio.Task[object]:
        if not self._intake_open:
            raise RuntimeError("intake closed")
        task = asyncio.ensure_future(asyncio.to_thread(self._runtime.process_request, request, task_id=task_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def submit_async(self, work: Callable[[], Awaitable[object]]) -> asyncio.Task[object]:
        if not self._intake_open:
            raise RuntimeError("intake closed")
        task = asyncio.ensure_future(work())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def stop_intake(self) -> None:
        self._intake_open = False

    def active_work_count(self) -> int:
        return sum(1 for task in self._tasks if not task.done())

    async def wait_for_idle(self) -> None:
        pending = [task for task in self._tasks if not task.done()]
        if pending:
            await asyncio.wait(pending)

    async def cancel_active_work(self) -> None:
        pending = [task for task in self._tasks if not task.done()]
        self.cancelled += len(pending)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _worker(data_dir: Path, counter: _Counter, *, idempotency_store) -> WorkerRuntime:  # type: ignore[no-untyped-def]
    from tests.unit.runtime.background_execution.reentry_admission_doubles import admission_kwargs

    kv = SqliteDistributedKVStore(data_dir / "worker_kv.sqlite3")
    registry = TaskRegistry()
    registry.register(TaskDefinition(task_name=TASK_NAME, payload_schema=dict, handler=counter.handler))
    execution_registry = TaskExecutionRegistry()
    registry.bind_execution_registry(execution_registry)
    deps = wire_background_execution_admission_dependencies(kv_store=kv)
    validate_background_execution_admission_durability(deps)  # fail-closed durability gate
    return WorkerRuntime(
        registry=registry,
        state_store=TaskStateStore(kv_store=kv, provider="sqlite"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="sqlite",
        idempotency_store=idempotency_store,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        **admission_kwargs(deps),  # pyright: ignore[reportArgumentType]
    )


def _request(key: str, run_id: str) -> TaskRequest:
    return TaskRequest(
        tenant_id=TENANT,
        run_id=run_id,
        task_name=TASK_NAME,
        payload=b"{}",
        idempotency_key=key,
    )


def _drain_profile(drain: float):  # type: ignore[no-untyped-def]
    return make_profile(
        runtime=JournalRuntime(),
        shutdown=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=drain,
            cancel_timeout_seconds=2.0,
            flush_timeout_seconds=1.0,
        ),
    )


def _ledger_status(db: Path) -> dict[str, str]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT schedule_id, status FROM scheduled_resumes").fetchall()
    return {sid: status for sid, status in rows}


async def test_xd05_shutdown_drains_real_worker_runtime_calls_and_restart_does_not_duplicate(
    tmp_path: Path,
) -> None:
    persistence = create_sqlite_runtime_persistence(data_dir=tmp_path / "sqlite")
    release = threading.Event()
    counter = _Counter(hold=release)
    runtime = _worker(tmp_path, counter, idempotency_store=persistence.idempotency_store)
    controller = _WorkerActiveWork(runtime)
    engine, publisher, _ = build_engine(
        definition_of(_drain_profile(15.0)), tmp_path / "host", active_work_controller=controller
    )
    await engine.start()

    first = controller.submit(_request("key-1", "run-1"), "run-1")
    second = controller.submit(_request("key-2", "run-2"), "run-2")
    assert await asyncio.to_thread(counter.entered.wait, _WAIT)
    assert controller.active_work_count() == 2

    stopper = asyncio.ensure_future(engine.stop(reason_code="xd05.drain"))
    await asyncio.sleep(0.2)
    assert not stopper.done(), "shutdown must wait for the in-flight worker calls"
    assert engine.accepts_new_work is False
    with pytest.raises(RuntimeError):
        controller.submit(_request("late", "run-3"), "run-3")  # intake closed first
    release.set()
    result = await asyncio.wait_for(stopper, timeout=_WAIT + 10)

    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    assert [r.status for r in await asyncio.gather(first, second)] == [TaskStatus.SUCCEEDED] * 2  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    assert controller.cancelled == 0 and controller.active_work_count() == 0
    execution = result.diagnostics.shutdown_execution
    assert execution is not None
    assert (execution.active_work_before, execution.active_work_after) == (2, 0)
    assert execution.timed_out is False and execution.forced is False
    phases = {record.phase: record for record in execution.phase_records}
    assert phases[HostedApplicationShutdownPhase.DRAIN].outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED
    assert HostedApplicationShutdownPhase.CANCEL not in phases
    assert len(publisher.of_type(HostedApplicationEventType.APPLICATION_STOPPED)) == 1
    exit_record = HostedApplicationExitClassifier().classify_terminal_result(
        result,
        application_id=APP_ID,
        instance_id="instance-001",
        profile_digest=engine.definition.profile_digest,
        occurred_at=SystemClock().now(),
        shutdown_execution=execution,
    )
    assert exit_record.exit_kind is HostedApplicationExitKind.CLEAN_STOP

    # Drained work is durably recorded in the worker state store.
    for run_id in ("run-1", "run-2"):
        assert runtime._state_store.get_status(tenant_id=TENANT, task_id=run_id) is TaskStatus.SUCCEEDED
    assert sorted(counter.calls) == ["key-1", "key-2"]

    # Restart: a fresh worker over the same durable stores redelivers both requests. Nothing re-runs.
    restarted_counter = _Counter()
    restarted = _worker(tmp_path, restarted_counter, idempotency_store=persistence.idempotency_store)
    redelivered = [
        restarted.process_request(_request("key-1", "run-1"), task_id="run-1"),  # same transport task
        restarted.process_request(_request("key-2", "run-9"), task_id="run-9"),  # new transport task, same key
    ]
    assert [r.status for r in redelivered] == [TaskStatus.SUCCEEDED] * 2
    assert restarted_counter.calls == [], "redelivery after restart must not duplicate the side effect"
    claim = persistence.idempotency_store.claim(
        tenant_id=TENANT, key=f"{TASK_NAME}:key-1", owner_id="probe", lease_seconds=5
    )
    assert claim.outcome is ClaimOutcome.REPLAY_COMPLETED  # the lease was released as COMPLETED


async def test_xd05_worker_call_outliving_the_drain_budget_is_fenced_not_duplicated(tmp_path: Path) -> None:
    persistence = create_sqlite_runtime_persistence(data_dir=tmp_path / "sqlite")
    release = threading.Event()
    counter = _Counter(hold=release)
    runtime = _worker(tmp_path, counter, idempotency_store=persistence.idempotency_store)
    controller = _WorkerActiveWork(runtime)
    engine, _, _ = build_engine(
        definition_of(_drain_profile(0.3)), tmp_path / "host", active_work_controller=controller
    )
    await engine.start()
    stuck = controller.submit(_request("key-stuck", "run-stuck"), "run-stuck")
    assert await asyncio.to_thread(counter.entered.wait, _WAIT)

    try:
        result = await asyncio.wait_for(engine.stop(reason_code="xd05.budget"), timeout=_WAIT)
        assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
        execution = result.diagnostics.shutdown_execution
        assert execution is not None and execution.forced is True
        phases = {record.phase: record for record in execution.phase_records}
        assert phases[HostedApplicationShutdownPhase.DRAIN].outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT
        assert HostedApplicationShutdownPhase.CANCEL in phases
        assert controller.cancelled == 1 and stuck.cancelled()

        # The abandoned worker thread still owns the ledger claim: a restarted worker is fenced
        # off the same idempotency key instead of running the side effect a second time.
        restarted_counter = _Counter()
        restarted = _worker(tmp_path, restarted_counter, idempotency_store=persistence.idempotency_store)
        probe = persistence.idempotency_store.claim(
            tenant_id=TENANT, key=f"{TASK_NAME}:key-stuck", owner_id="restart", lease_seconds=5
        )
        assert probe.outcome is ClaimOutcome.BLOCKED_ACTIVE
        with pytest.raises(Exception, match="already in progress"):
            await asyncio.to_thread(restarted.process_request, _request("key-stuck", "run-x"), task_id="run-x")
        assert restarted_counter.calls == []
    finally:
        release.set()
    # Once the abandoned call finishes, the lease is released as COMPLETED and replays.
    replay = None
    for _ in range(100):
        replay = persistence.idempotency_store.claim(
            tenant_id=TENANT, key=f"{TASK_NAME}:key-stuck", owner_id="probe-2", lease_seconds=5
        )
        if replay.outcome is ClaimOutcome.REPLAY_COMPLETED:
            break
        await asyncio.sleep(0.1)
    assert replay is not None and replay.outcome is ClaimOutcome.REPLAY_COMPLETED
    assert counter.calls == ["key-stuck"]


class _GatedResume:
    """Scheduler resume executor (the real port) that can hold the resume open."""

    def __init__(self, gate_seconds: float) -> None:
        self.gate_seconds = gate_seconds
        self.entered = asyncio.Event()
        self.resumed: list[str] = []

    async def resume_task(self, task, *, checkpoint):  # type: ignore[no-untyped-def]
        self.entered.set()
        await asyncio.sleep(self.gate_seconds)
        self.resumed.append(str(task.metadata.get("schedule_id")))
        return ok_task_result(checkpoint.task_id)


def _schedule_due(persistence):  # type: ignore[no-untyped-def]
    store = persistence.task_checkpoint_store
    checkpoint = paused_checkpoint()
    store.save(checkpoint)
    entry = store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=(T0 - timedelta(minutes=5)).isoformat(),
        )
    )
    return store, entry


def _scheduler(store, executor, owner: str) -> LongRunningScheduler:  # type: ignore[no-untyped-def]
    return LongRunningScheduler(store, executor, schedule_store=store, ledger=store, owner_id=owner, lease_seconds=1)


async def test_xd05_shutdown_drains_an_in_flight_scheduled_resume_and_releases_the_lease(tmp_path: Path) -> None:
    persistence = create_sqlite_runtime_persistence(data_dir=tmp_path / "sqlite")
    store, entry = _schedule_due(persistence)
    db = persistence.paths.task_checkpoints
    executor = _GatedResume(gate_seconds=0.4)
    scheduler = _scheduler(store, executor, "sched-1")
    controller = _WorkerActiveWork(_worker(tmp_path, _Counter(), idempotency_store=persistence.idempotency_store))
    engine, _, _ = build_engine(
        definition_of(_drain_profile(15.0)), tmp_path / "host", active_work_controller=controller
    )
    await engine.start()

    tick = controller.submit_async(lambda: scheduler.tick(now=T0))
    await asyncio.wait_for(executor.entered.wait(), timeout=_WAIT)
    assert _ledger_status(db)[entry.schedule_id] == "running"  # claim held (leased) mid-resume

    result = await asyncio.wait_for(engine.stop(reason_code="xd05.sched.drain"), timeout=_WAIT)
    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    assert tick.result() == 1 and controller.cancelled == 0
    assert executor.resumed == [entry.schedule_id]
    assert _ledger_status(db)[entry.schedule_id] == "completed"  # lease released on drain

    # Restart with a different owner: the completed occurrence is never fired again.
    restarted_executor = _GatedResume(gate_seconds=0.0)
    restarted = _scheduler(store, restarted_executor, "sched-2")
    assert await restarted.tick(now=T0 + timedelta(seconds=30)) == 0
    assert restarted_executor.resumed == []


async def test_xd05_forced_shutdown_leaves_a_fenced_claim_that_restart_never_refires(tmp_path: Path) -> None:
    persistence = create_sqlite_runtime_persistence(data_dir=tmp_path / "sqlite")
    store, entry = _schedule_due(persistence)
    db = persistence.paths.task_checkpoints
    executor = _GatedResume(gate_seconds=60.0)
    scheduler = _scheduler(store, executor, "sched-1")
    controller = _WorkerActiveWork(_worker(tmp_path, _Counter(), idempotency_store=persistence.idempotency_store))
    engine, _, _ = build_engine(
        definition_of(_drain_profile(0.3)), tmp_path / "host", active_work_controller=controller
    )
    await engine.start()
    tick = controller.submit_async(lambda: scheduler.tick(now=T0))
    await asyncio.wait_for(executor.entered.wait(), timeout=_WAIT)

    result = await asyncio.wait_for(engine.stop(reason_code="xd05.sched.force"), timeout=_WAIT)
    execution = result.diagnostics.shutdown_execution
    assert execution is not None and execution.forced is True
    assert tick.cancelled() and controller.cancelled == 1
    assert _ledger_status(db)[entry.schedule_id] == "running"  # orphaned claim, lease not yet expired

    # The store judges lease expiry on the wall clock; wait it out, then a restarted scheduler
    # fences the orphan (uncertain) and never refires it.
    await asyncio.sleep(1.5)
    restarted_executor = _GatedResume(gate_seconds=0.0)
    restarted = _scheduler(store, restarted_executor, "sched-2")
    assert await restarted.tick(now=T0 + timedelta(seconds=60)) == 0
    assert restarted_executor.resumed == []
    assert _ledger_status(db)[entry.schedule_id] == "uncertain"

    stale = ScheduledResumeClaim(
        schedule_id=entry.schedule_id,
        owner_id="sched-1",
        lease_expires_at=T0,
        fence=1,
        entry=entry,
    )
    with pytest.raises(StaleClaimError):
        store.complete_claim(stale)  # the old owner's late completion is rejected
    assert _ledger_status(db)[entry.schedule_id] == "uncertain"
